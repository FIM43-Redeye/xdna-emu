//! Buffer Descriptor (BD) parsing and types for AIE-ML DMA.
//!
//! This module provides clean BD parsing from register layouts for all three tile types:
//! - Compute tile: 6 registers per BD, 16 BDs, 14-bit word address
//! - MemTile: 8 registers per BD, 48 BDs, 19-bit word address
//! - Shim tile: 8 registers per BD, 16 BDs, 46-bit word address
//!
//! All addresses and stepsizes are in 32-bit WORD units (hardware native).
//! Stepsizes in hardware are stored as (actual - 1); we convert to actual values.
//!
//! Reference: AMD AM020/AM025 and docs/dma-reference.md

use xdna_archspec::types::TileKind;

/// BD spacing: 32 bytes (0x20) between BDs for all tile types.
/// Consistent across compute, memtile, and shim; also derivable from regdb.
pub const BD_SPACING: u64 = 0x20;

/// Parsed Buffer Descriptor fields (common to all tile types).
///
/// All addresses and stepsizes are in 32-bit WORD units.
/// Stepsizes are converted to actual values (stored + 1).
#[derive(Debug, Clone, Default)]
pub struct BufferDescriptor {
    /// BD is valid and can be used
    pub valid: bool,

    /// Base address in 32-bit word units
    pub base_addr_words: u64,

    /// Transfer length in 32-bit words
    pub length_words: u32,

    // Dimensional addressing (all stepsizes are actual values)
    /// Dimension 0 stepsize in words (actual, not stored-1)
    pub d0_stepsize: u32,
    /// Dimension 0 wrap count (0 = no wrap)
    pub d0_wrap: u16,

    /// Dimension 1 stepsize in words
    pub d1_stepsize: u32,
    /// Dimension 1 wrap count
    pub d1_wrap: u16,

    /// Dimension 2 stepsize in words
    pub d2_stepsize: u32,
    /// Dimension 2 wrap count
    pub d2_wrap: u16,

    /// Dimension 3 stepsize in words (MemTile only)
    pub d3_stepsize: u32,

    // Iteration (outermost loop)
    /// Iteration stepsize in words (actual)
    pub iteration_stepsize: u32,
    /// Iteration wrap (actual, 1-64)
    pub iteration_wrap: u8,
    /// Current iteration counter
    pub iteration_current: u8,

    // Lock configuration
    /// Enable lock acquire before transfer
    pub lock_acq_enable: bool,
    /// Lock ID to acquire (4-bit for Compute/Shim, 8-bit for MemTile)
    pub lock_acq_id: u8,
    /// Lock acquire value (signed: >=0 means exact match, <0 means GE)
    pub lock_acq_value: i8,
    /// Lock ID to release after transfer
    pub lock_rel_id: u8,
    /// Lock release delta (signed, 0 = no release)
    pub lock_rel_value: i8,

    // BD chaining
    /// Continue with next BD after completion
    pub use_next_bd: bool,
    /// Next BD ID to chain to
    pub next_bd: u8,

    // Packet mode (MM2S only)
    /// Enable packet header insertion
    pub enable_packet: bool,
    /// Packet ID (5 bits)
    pub packet_id: u8,
    /// Packet type (3 bits)
    pub packet_type: u8,
    /// Out-of-order BD ID (6 bits)
    pub ooo_bd_id: u8,
    /// Suppress TLAST at end of transfer
    pub tlast_suppress: bool,

    // Compression (optional)
    /// Enable compression (MM2S) or decompression (S2MM)
    pub compression_enable: bool,

    // MemTile-specific: zero padding (MM2S only)
    //
    // Unit convention: D0 padding is in 32-bit WORD units (matching d0_wrap).
    // D1/D2 padding is in iteration counts (matching d1_wrap/d2_wrap).
    //
    // NOTE: mlir-aie has two paths that program these fields:
    // - CDO path (AIERT.cpp): converts D0 padding from elements to words
    // - NPU task path (AIEDMATasksToNPU.cpp): writes raw element counts
    // The CDO path is standard; the NPU task path inconsistency may be
    // an mlir-aie bug. Our emulator treats these as word units per aie-rt.
    /// D0 zeros before (in 32-bit word units)
    pub d0_zero_before: u8,
    /// D0 zeros after (in 32-bit word units)
    pub d0_zero_after: u8,
    /// D1 zeros before (iteration count)
    pub d1_zero_before: u8,
    /// D1 zeros after (iteration count)
    pub d1_zero_after: u8,
    /// D2 zeros before (iteration count)
    pub d2_zero_before: u8,
    /// D2 zeros after (iteration count)
    pub d2_zero_after: u8,

    // Shim-specific: AXI parameters
    /// Burst length (0=64B, 1=128B, 2=256B)
    pub burst_length: u8,
    /// AXI SMID
    pub smid: u8,
    /// AXI cache attributes
    pub axcache: u8,
    /// AXI QoS
    pub axqos: u8,
    /// Secure access flag
    pub secure_access: bool,
}

impl BufferDescriptor {
    /// Create a new empty/invalid BD
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse BD from register words based on tile type.
    ///
    /// # Arguments
    /// * `words` - Slice of 6-8 u32 register values (tile-type dependent)
    /// * `tile_type` - Type of tile this BD belongs to
    ///
    /// # Panics
    /// Panics if `words` slice is too short for the tile type.
    pub fn from_registers(words: &[u32], tile_kind: TileKind) -> Self {
        match tile_kind {
            TileKind::Compute => Self::parse_compute(words),
            TileKind::Mem => Self::parse_memtile(words),
            TileKind::ShimNoc | TileKind::ShimPl => Self::parse_shim(words),
        }
    }

    /// Parse Compute tile BD (6 registers per BD).
    ///
    /// Field layouts are derived from the register database (AM025 memory module).
    /// All bit positions come from `BdFieldLayout::from_regdb()` rather than
    /// hardcoded shift/mask constants. This matches the pattern used by `parse_shim()`.
    fn parse_compute(words: &[u32]) -> Self {
        let expected = bd_register_count(TileKind::Compute);
        assert!(words.len() >= expected, "Compute BD needs {} registers", expected);

        let lay = &crate::device::regdb::device_reg_layout().memory_bd;

        let w0 = words[0];
        let w1 = words[1];
        let w2 = words[2];
        let w3 = words[3];
        let w4 = words[4];
        let w5 = words[5];

        Self {
            // BD_0: Base_Address, Buffer_Length
            base_addr_words: lay.base_address.extract(w0) as u64,
            length_words: lay.buffer_length.extract(w0),

            // BD_1: compression, packet control
            compression_enable: lay.enable_compression.extract_bool(w1),
            enable_packet: lay.enable_packet.extract_bool(w1),
            ooo_bd_id: lay.out_of_order_bd_id.extract(w1) as u8,
            packet_id: lay.packet_id.extract(w1) as u8,
            packet_type: lay.packet_type.extract(w1) as u8,

            // BD_2: stepsizes (stored as actual-1, convert to actual)
            d1_stepsize: lay.d1_stepsize.extract(w2) + 1,
            d0_stepsize: lay.d0_stepsize.extract(w2) + 1,

            // BD_3: wrap counts and D2 stepsize
            d1_wrap: lay.d1_wrap.extract(w3) as u16,
            d0_wrap: lay.d0_wrap.extract(w3) as u16,
            d2_stepsize: lay.d2_stepsize.extract(w3) + 1,

            // BD_4: iteration control
            iteration_current: lay.iteration_current.extract(w4) as u8,
            iteration_wrap: (lay.iteration_wrap.extract(w4) + 1) as u8, // stored as actual-1
            iteration_stepsize: lay.iteration_stepsize.extract(w4) + 1,

            // BD_5: TLAST, chaining, locks
            tlast_suppress: lay.tlast_suppress.extract_bool(w5),
            next_bd: lay.next_bd.extract(w5) as u8,
            use_next_bd: lay.use_next_bd.extract_bool(w5),
            valid: lay.valid_bd.extract_bool(w5),
            lock_rel_value: sign_extend_7bit(lay.lock_rel_value.extract(w5) as u8),
            lock_rel_id: lay.lock_rel_id.extract(w5) as u8,
            lock_acq_enable: lay.lock_acq_enable.extract_bool(w5),
            lock_acq_value: sign_extend_7bit(lay.lock_acq_value.extract(w5) as u8),
            lock_acq_id: lay.lock_acq_id.extract(w5) as u8,

            // Compute tile BDs have D2_Stepsize but no D2_Wrap register
            // field. Like shim tiles, the D2 iteration count is implicit:
            // d2_wrap = buffer_length / (d0_size * d1_size).
            d3_stepsize: 0,
            d2_wrap: {
                let d0_sz = lay.d0_wrap.extract(w3) as u32;
                let d1_sz = lay.d1_wrap.extract(w3) as u32;
                let buf_len = lay.buffer_length.extract(w0);
                let d2_step = lay.d2_stepsize.extract(w3) + 1;
                if d0_sz > 0 && d1_sz > 0 && d2_step > 0 {
                    let page_size = d0_sz * d1_sz;
                    if page_size > 0 && buf_len > page_size {
                        (buf_len / page_size) as u16
                    } else {
                        0
                    }
                } else {
                    0
                }
            },
            d0_zero_before: 0,
            d0_zero_after: 0,
            d1_zero_before: 0,
            d1_zero_after: 0,
            d2_zero_before: 0,
            d2_zero_after: 0,
            burst_length: 0,
            smid: 0,
            axcache: 0,
            axqos: 0,
            secure_access: false,
        }
    }

    /// Parse MemTile BD (8 registers per BD).
    ///
    /// Field layouts are derived from the register database (AM025 memory_tile
    /// module). All bit positions come from `MemTileBdFieldLayout::from_regdb()`
    /// rather than hardcoded shift/mask constants.
    fn parse_memtile(words: &[u32]) -> Self {
        let expected = bd_register_count(TileKind::Mem);
        assert!(words.len() >= expected, "MemTile BD needs {} registers", expected);

        let lay = &crate::device::regdb::device_reg_layout().memtile_bd;

        let w0 = words[0];
        let w1 = words[1];
        let w2 = words[2];
        let w3 = words[3];
        let w4 = words[4];
        let w5 = words[5];
        let w6 = words[6];
        let w7 = words[7];

        Self {
            // BD_0: packet control and buffer length
            enable_packet: lay.enable_packet.extract_bool(w0),
            packet_type: lay.packet_type.extract(w0) as u8,
            packet_id: lay.packet_id.extract(w0) as u8,
            ooo_bd_id: lay.out_of_order_bd_id.extract(w0) as u8,
            length_words: lay.buffer_length.extract(w0),

            // BD_1: zero-padding, chaining, base address
            d0_zero_before: lay.d0_zero_before.extract(w1) as u8,
            next_bd: lay.next_bd.extract(w1) as u8,
            use_next_bd: lay.use_next_bd.extract_bool(w1),
            base_addr_words: lay.base_address.extract(w1) as u64,

            // BD_2: TLAST and D0 addressing
            tlast_suppress: lay.tlast_suppress.extract_bool(w2),
            d0_wrap: lay.d0_wrap.extract(w2) as u16,
            d0_stepsize: lay.d0_stepsize.extract(w2) + 1, // stored as actual-1

            // BD_3: D1 zero-padding and addressing
            d1_zero_before: lay.d1_zero_before.extract(w3) as u8,
            d1_wrap: lay.d1_wrap.extract(w3) as u16,
            d1_stepsize: lay.d1_stepsize.extract(w3) + 1,

            // BD_4: compression, D2 zero-padding and addressing
            compression_enable: lay.enable_compression.extract_bool(w4),
            d2_zero_before: lay.d2_zero_before.extract(w4) as u8,
            d2_wrap: lay.d2_wrap.extract(w4) as u16,
            d2_stepsize: lay.d2_stepsize.extract(w4) + 1,

            // BD_5: zero-after padding and D3 stepsize
            d2_zero_after: lay.d2_zero_after.extract(w5) as u8,
            d1_zero_after: lay.d1_zero_after.extract(w5) as u8,
            d0_zero_after: lay.d0_zero_after.extract(w5) as u8,
            d3_stepsize: lay.d3_stepsize.extract(w5) + 1, // MemTile has D3

            // BD_6: iteration control
            iteration_current: lay.iteration_current.extract(w6) as u8,
            iteration_wrap: (lay.iteration_wrap.extract(w6) + 1) as u8, // stored as actual-1
            iteration_stepsize: lay.iteration_stepsize.extract(w6) + 1,

            // BD_7: validity and lock synchronization
            valid: lay.valid_bd.extract_bool(w7),
            lock_rel_value: sign_extend_7bit(lay.lock_rel_value.extract(w7) as u8),
            lock_rel_id: lay.lock_rel_id.extract(w7) as u8, // 8 bits for MemTile
            lock_acq_enable: lay.lock_acq_enable.extract_bool(w7),
            lock_acq_value: sign_extend_7bit(lay.lock_acq_value.extract(w7) as u8),
            lock_acq_id: lay.lock_acq_id.extract(w7) as u8, // 8 bits for MemTile

            // Not used in MemTile
            burst_length: 0,
            smid: 0,
            axcache: 0,
            axqos: 0,
            secure_access: false,
        }
    }

    /// Parse Shim tile BD (8 registers per BD).
    ///
    /// Field layouts are derived from the register database (AM025 shim module).
    /// The shim BD is notable for its full 32-bit buffer length, 46-bit word
    /// address (split across two registers), and AXI parameters (burst length,
    /// SMID, AxCache, AxQoS).
    fn parse_shim(words: &[u32]) -> Self {
        let expected = bd_register_count(TileKind::ShimNoc);
        assert!(words.len() >= expected, "Shim BD needs {} registers", expected);

        let lay = &crate::device::regdb::device_reg_layout().shim_bd;

        let w0 = words[0];
        let w1 = words[1];
        let w2 = words[2];
        let w3 = words[3];
        let w4 = words[4];
        let w5 = words[5];
        let w6 = words[6];
        let w7 = words[7];

        // 46-bit word address from low and high parts
        let addr_low = lay.base_address_low.extract(w1) as u64; // 30 bits
        let addr_high = lay.base_address_high.extract(w2) as u64; // 16 bits
        let base_addr = addr_low | (addr_high << 30);

        Self {
            // BD_0: full 32-bit buffer length for DDR transfers
            length_words: lay.buffer_length.extract(w0),

            // BD_1 + BD_2: 46-bit word address
            base_addr_words: base_addr,

            // BD_2: packet control
            enable_packet: lay.enable_packet.extract_bool(w2),
            ooo_bd_id: lay.out_of_order_bd_id.extract(w2) as u8,
            packet_id: lay.packet_id.extract(w2) as u8,
            packet_type: lay.packet_type.extract(w2) as u8,

            // BD_3: secure access + D0
            secure_access: lay.secure_access.extract_bool(w3),
            d0_wrap: lay.d0_wrap.extract(w3) as u16,
            d0_stepsize: lay.d0_stepsize.extract(w3) + 1, // stored as actual-1

            // BD_4: burst length + D1
            burst_length: lay.burst_length.extract(w4) as u8,
            d1_wrap: lay.d1_wrap.extract(w4) as u16,
            d1_stepsize: lay.d1_stepsize.extract(w4) + 1,

            // BD_5: AXI parameters + D2
            smid: lay.smid.extract(w5) as u8,
            axcache: lay.axcache.extract(w5) as u8,
            axqos: lay.axqos.extract(w5) as u8,
            d2_stepsize: lay.d2_stepsize.extract(w5) + 1,

            // BD_6: iteration control
            iteration_current: lay.iteration_current.extract(w6) as u8,
            iteration_wrap: (lay.iteration_wrap.extract(w6) + 1) as u8, // stored as actual-1
            iteration_stepsize: lay.iteration_stepsize.extract(w6) + 1,

            // BD_7: locks and chaining
            tlast_suppress: lay.tlast_suppress.extract_bool(w7),
            next_bd: lay.next_bd.extract(w7) as u8,
            use_next_bd: lay.use_next_bd.extract_bool(w7),
            valid: lay.valid_bd.extract_bool(w7),
            lock_rel_value: sign_extend_7bit(lay.lock_rel_value.extract(w7) as u8),
            lock_rel_id: lay.lock_rel_id.extract(w7) as u8,
            lock_acq_enable: lay.lock_acq_enable.extract_bool(w7),
            lock_acq_value: sign_extend_7bit(lay.lock_acq_value.extract(w7) as u8),
            lock_acq_id: lay.lock_acq_id.extract(w7) as u8,

            // Shim BDs have no D2_Wrap register field. On real hardware,
            // the D2 iteration count is implicit: the DMA transfers
            // Buffer_Length words total, using D0 x D1 addressing per
            // "page" and stepping by D2_Stepsize between pages. The D2
            // wrap count is: buffer_length / (d0_size * d1_size).
            // When D0 and D1 are both simple (wrap=0), D2 is unused.
            d3_stepsize: 0,
            d2_wrap: {
                let d0_sz = lay.d0_wrap.extract(w3) as u32;
                let d1_sz = lay.d1_wrap.extract(w4) as u32;
                let buf_len = lay.buffer_length.extract(w0);
                let d2_step = lay.d2_stepsize.extract(w5) + 1;
                if d0_sz > 0 && d1_sz > 0 && d2_step > 0 {
                    let page_size = d0_sz * d1_sz;
                    if page_size > 0 && buf_len > page_size {
                        (buf_len / page_size) as u16
                    } else {
                        0
                    }
                } else {
                    0
                }
            },
            compression_enable: false,
            d0_zero_before: 0,
            d0_zero_after: 0,
            d1_zero_before: 0,
            d1_zero_after: 0,
            d2_zero_before: 0,
            d2_zero_after: 0,
        }
    }

    /// Parse BD from memory at the given BD slot.
    ///
    /// # Arguments
    /// * `memory` - Tile memory slice
    /// * `bd_id` - BD slot index
    /// * `tile_type` - Type of tile
    pub fn from_memory(memory: &[u8], bd_id: u8, tile_kind: TileKind) -> Self {
        let base = bd_base_address(tile_kind);
        let offset = (base + bd_id as u64 * BD_SPACING) as usize;
        let reg_count = bd_register_count(tile_kind);

        // Read register words from memory
        let mut words = Vec::with_capacity(reg_count);
        for i in 0..reg_count {
            let addr = offset + i * 4;
            if addr + 4 <= memory.len() {
                let word =
                    u32::from_le_bytes([memory[addr], memory[addr + 1], memory[addr + 2], memory[addr + 3]]);
                words.push(word);
            } else {
                words.push(0);
            }
        }

        Self::from_registers(&words, tile_kind)
    }

    /// Convert byte address to word address
    pub fn byte_to_word_addr(byte_addr: u64) -> u64 {
        byte_addr / 4
    }

    /// Convert word address to byte address
    pub fn word_to_byte_addr(word_addr: u64) -> u64 {
        word_addr * 4
    }

    /// Get the byte address for this BD
    pub fn base_addr_bytes(&self) -> u64 {
        Self::word_to_byte_addr(self.base_addr_words)
    }

    /// Get the transfer length in bytes
    pub fn length_bytes(&self) -> u64 {
        self.length_words as u64 * 4
    }

    /// Convert to the runtime BdConfig used by the DMA engine.
    ///
    /// BufferDescriptor stores hardware-native units (word addresses, word
    /// strides, actual values for stepsizes and wraps). BdConfig stores
    /// byte-addressed values for the emulator's address generator, and uses
    /// the "stored - 1" convention for iteration fields.
    pub fn to_bd_config(&self) -> super::BdConfig {
        use super::{BdConfig, DimensionConfig, IterationConfig};

        // Convert word address to byte address
        let base_addr = self.base_addr_words * 4;
        let length = self.length_words * 4;

        // Convert word strides to byte strides: stride_bytes = stride_words * 4
        // d0_wrap of 0 means simple contiguous transfer -- use length_words as size
        let d0_size = if self.d0_wrap == 0 {
            self.length_words
        } else {
            self.d0_wrap as u32
        };
        let d0_stride = (self.d0_stepsize as i32) * 4;

        let d1_size = self.d1_wrap as u32;
        let d1_stride = (self.d1_stepsize as i32) * 4;

        let d2_size = self.d2_wrap as u32;
        let d2_stride = (self.d2_stepsize as i32) * 4;

        let d3_stride = (self.d3_stepsize as i32) * 4;

        // MemTile BDs support 4 explicit dimensions per aie-rt
        // `AieMlMemTileMultiDimProp`: D0/D1/D2 have (wrap, stepsize)
        // register fields, but D3 has only a stepsize -- its size is
        // implicit from buffer_length / (d0_size * d1_size * d2_size).
        // This mirrors the derivation we already do for d2_wrap on
        // compute/shim BDs (which lack a d2_wrap register). Without
        // this, a BD configured with 4D addressing underruns: the
        // address generator finishes after d0*d1*d2 words and
        // `current()` returns the stuck-at-last address for every
        // subsequent word (observed as repeated `0x27` in
        // dma_complex_dims).
        //
        // For simple transfers where length == d0*d1*d2, d3 stays 0
        // (effective 1) so the dim product is unchanged. Guarded by
        // d3_stepsize > 0 to restrict this to MemTile BDs: compute and
        // shim parsers explicitly set d3_stepsize = 0 since those tile
        // types have no D3 register field.
        let d3_size = if self.d3_stepsize > 0 {
            let d0_eff = if d0_size == 0 { 1 } else { d0_size };
            let d1_eff = if d1_size == 0 { 1 } else { d1_size };
            let d2_eff = if d2_size == 0 { 1 } else { d2_size };
            let product = d0_eff * d1_eff * d2_eff;
            if product > 0 && self.length_words > product {
                self.length_words / product
            } else {
                0
            }
        } else {
            0
        };

        // IterationConfig uses stored-1 convention: wrap=0 means 1 iteration,
        // stepsize is also stored-1. BufferDescriptor already has actual values
        // (e.g., iteration_wrap=3 means 3 iterations). Convert back.
        let iteration = IterationConfig {
            current: self.iteration_current,
            wrap: self.iteration_wrap.saturating_sub(1),
            stepsize: self.iteration_stepsize.saturating_sub(1) as u16,
        };

        // Lock configuration: acquire if enabled, release if value is non-zero
        let acquire_lock = if self.lock_acq_enable {
            Some(self.lock_acq_id)
        } else {
            None
        };
        let release_lock = if self.lock_rel_value != 0 {
            Some(self.lock_rel_id)
        } else {
            None
        };

        // BD chaining: chain if use_next_bd is set AND the BD is valid
        let next_bd = if self.use_next_bd {
            Some(self.next_bd)
        } else {
            None
        };

        BdConfig {
            base_addr,
            length,
            d0: DimensionConfig { size: d0_size, stride: d0_stride },
            d1: DimensionConfig { size: d1_size, stride: d1_stride },
            d2: DimensionConfig { size: d2_size, stride: d2_stride },
            d3: DimensionConfig { size: d3_size, stride: d3_stride },
            iteration,
            compression_enable: self.compression_enable,
            zero_padding: {
                let pad = super::addressing::ZeroPadConfig {
                    d0_before: self.d0_zero_before,
                    d0_after: self.d0_zero_after,
                    d1_before: self.d1_zero_before,
                    d1_after: self.d1_zero_after,
                    d2_before: self.d2_zero_before,
                    d2_after: self.d2_zero_after,
                };
                // Validate MemTile padding constraints per aie-rt
                // _XAieMl_DmaMemTileCheckPaddingConfig(). Uses raw wrap values
                // (0 = simple/1 iteration) to check propagation rules.
                pad.validate_padding(self.d0_wrap, self.d1_wrap, self.d2_wrap);
                pad
            },
            enable_packet: self.enable_packet,
            packet_id: self.packet_id,
            packet_type: self.packet_type,
            out_of_order_bd_id: self.ooo_bd_id,
            out_of_order: false, // S2MM-only, set by channel config
            tlast_suppress: self.tlast_suppress,
            acquire_lock,
            acquire_value: self.lock_acq_value,
            release_lock,
            release_value: self.lock_rel_value,
            next_bd,
            valid: self.valid,
        }
    }
}

/// Get BD base address for a tile type (from register database).
pub fn bd_base_address(tile_kind: TileKind) -> u64 {
    let lay = crate::device::regdb::device_reg_layout();
    match tile_kind {
        TileKind::Compute => lay.memory_bd_base as u64,
        TileKind::Mem => lay.memtile_bd_base as u64,
        TileKind::ShimNoc | TileKind::ShimPl => lay.shim_bd_base as u64,
    }
}

/// Get number of registers per BD for a tile type (derived from regdb).
///
/// Counts DMA_BD0_N registers in the register database for each module type.
/// The regdb computes this at init time and stores in DeviceRegLayout.
pub fn bd_register_count(tile_kind: TileKind) -> usize {
    let lay = crate::device::regdb::device_reg_layout();
    match tile_kind {
        TileKind::Compute => lay.memory_bd_words,
        TileKind::Mem => lay.memtile_bd_words,
        TileKind::ShimNoc | TileKind::ShimPl => lay.shim_bd_words,
    }
}

/// Sign-extend a 7-bit value to i8
fn sign_extend_7bit(val: u8) -> i8 {
    if val & 0x40 != 0 {
        // Negative: extend sign
        (val | 0x80) as i8
    } else {
        val as i8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sign_extend_7bit() {
        assert_eq!(sign_extend_7bit(0), 0);
        assert_eq!(sign_extend_7bit(1), 1);
        assert_eq!(sign_extend_7bit(63), 63);
        assert_eq!(sign_extend_7bit(64), -64); // 0x40 -> -64
        assert_eq!(sign_extend_7bit(127), -1); // 0x7F -> -1
    }

    #[test]
    fn test_compute_bd_parsing() {
        // Sample BD registers for compute tile
        let words = [
            0x0004_1000, // BD_0: base=0x10, length=0x1000
            0x4000_0000, // BD_1: packet enabled
            0x0000_0001, // BD_2: d0_step=2, d1_step=1
            0x0000_0000, // BD_3: no wrap
            0x0000_0000, // BD_4: no iteration
            0x0200_0000, // BD_5: valid=1
        ];

        let bd = BufferDescriptor::from_registers(&words, TileKind::Compute);

        assert!(bd.valid);
        assert_eq!(bd.base_addr_words, 0x10);
        assert_eq!(bd.length_words, 0x1000);
        assert!(bd.enable_packet);
        assert_eq!(bd.d0_stepsize, 2); // stored 1 + 1 = 2
        assert_eq!(bd.d1_stepsize, 1); // stored 0 + 1 = 1
    }

    #[test]
    fn test_memtile_bd_parsing() {
        // Sample BD registers for memtile
        // BD_1 layout: D0_Zero_Before[31:26], Next_BD[25:20], Use_Next_BD[19], Base_Address[18:0]
        // For next_bd=2: bit 21 (in Next_BD field) = 0x00200000
        // For use_next=1: bit 19 = 0x00080000
        // For addr=0x100: bits 18:0 = 0x00000100
        // Combined: 0x00280100
        let words = [
            0x8000_0400, // BD_0: packet=1, length=0x400
            0x0028_0100, // BD_1: next_bd=2, use_next=1, addr=0x100
            0x0000_0003, // BD_2: d0_step=4
            0x0000_0000, // BD_3
            0x0000_0000, // BD_4
            0x0000_0000, // BD_5
            0x0000_0000, // BD_6
            0x8000_0000, // BD_7: valid=1
        ];

        let bd = BufferDescriptor::from_registers(&words, TileKind::Mem);

        assert!(bd.valid);
        assert!(bd.enable_packet);
        assert_eq!(bd.length_words, 0x400);
        assert_eq!(bd.base_addr_words, 0x100);
        assert!(bd.use_next_bd);
        assert_eq!(bd.next_bd, 2);
        assert_eq!(bd.d0_stepsize, 4);
    }

    #[test]
    fn test_shim_bd_parsing() {
        // Sample BD registers for shim tile
        let words = [
            0x0000_1000, // BD_0: length=0x1000
            0x0000_0400, // BD_1: addr_low=0x100
            0x0000_0000, // BD_2: addr_high=0
            0x0010_0003, // BD_3: d0_wrap=1, d0_step=4
            0x0000_0000, // BD_4
            0x0000_0000, // BD_5
            0x0000_0000, // BD_6
            0x0200_0000, // BD_7: valid=1
        ];

        let bd = BufferDescriptor::from_registers(&words, TileKind::ShimNoc);

        assert!(bd.valid);
        assert_eq!(bd.length_words, 0x1000);
        assert_eq!(bd.base_addr_words, 0x100);
        assert_eq!(bd.d0_wrap, 1);
        assert_eq!(bd.d0_stepsize, 4);
    }

    /// Cross-validate regdb-driven parse_compute() against AM025 bit positions.
    ///
    /// Constructs BD words where every field has a distinct, non-zero value
    /// placed at exact AM025 bit positions, then verifies that parse_compute()
    /// (which uses regdb extraction) produces the expected values. This catches
    /// any mismatch between JSON field definitions and the AM025 register spec.
    #[test]
    fn test_compute_bd_cross_validation() {
        // Exact bit positions from AM025 / aie_registers_aie2.json:
        //
        // DMA_BD0_0: Buffer_Length[13:0], Base_Address[27:14]
        // DMA_BD0_1: Packet_Type[18:16], Packet_ID[23:19], OOO_BD_ID[29:24],
        //            Enable_Packet[30], Enable_Compression[31]
        // DMA_BD0_2: D0_Stepsize[12:0], D1_Stepsize[25:13]
        // DMA_BD0_3: D2_Stepsize[12:0], D0_Wrap[20:13], D1_Wrap[28:21]
        // DMA_BD0_4: Iteration_Stepsize[12:0], Iteration_Wrap[18:13],
        //            Iteration_Current[24:19]
        // DMA_BD0_5: Lock_Acq_ID[3:0], Lock_Acq_Value[11:5], Lock_Acq_Enable[12],
        //            Lock_Rel_ID[16:13], Lock_Rel_Value[24:18], Valid_BD[25],
        //            Use_Next_BD[26], Next_BD[30:27], TLAST_Suppress[31]

        let w0: u32 = (0x1AB << 14) | 0x2345;
        let w1: u32 = (1 << 31) | (1 << 30) | (0x15 << 24) | (0x0A << 19) | (0x5 << 16);
        let w2: u32 = (99 << 13) | 49; // D1_Step[25:13]=99, D0_Step[12:0]=49
        let w3: u32 = (7 << 21) | (3 << 13) | 19; // D1_Wrap[28:21]=7, D0_Wrap[20:13]=3, D2_Step[12:0]=19
        let w4: u32 = (5 << 19) | (2 << 13) | 9; // Iter_Cur[24:19]=5, Iter_Wrap[18:13]=2, Iter_Step[12:0]=9
        let w5: u32 =
            (1 << 31) | (5 << 27) | (1 << 26) | (1 << 25) | (3 << 18) | (2 << 13) | (1 << 12) | (1 << 5) | 7;

        let words = [w0, w1, w2, w3, w4, w5];
        let bd = BufferDescriptor::from_registers(&words, TileKind::Compute);

        // Word 0
        assert_eq!(bd.base_addr_words, 0x1AB, "base_addr_words");
        assert_eq!(bd.length_words, 0x2345, "length_words");

        // Word 1
        assert!(bd.compression_enable, "compression_enable");
        assert!(bd.enable_packet, "enable_packet");
        assert_eq!(bd.ooo_bd_id, 0x15, "ooo_bd_id");
        assert_eq!(bd.packet_id, 0x0A, "packet_id");
        assert_eq!(bd.packet_type, 0x5, "packet_type");

        // Word 2 (stepsizes stored as actual-1, parser adds 1)
        assert_eq!(bd.d0_stepsize, 50, "d0_stepsize (stored 49 + 1)");
        assert_eq!(bd.d1_stepsize, 100, "d1_stepsize (stored 99 + 1)");

        // Word 3
        assert_eq!(bd.d0_wrap, 3, "d0_wrap");
        assert_eq!(bd.d1_wrap, 7, "d1_wrap");
        assert_eq!(bd.d2_stepsize, 20, "d2_stepsize (stored 19 + 1)");

        // Word 4
        assert_eq!(bd.iteration_current, 5, "iteration_current");
        assert_eq!(bd.iteration_wrap, 3, "iteration_wrap (stored 2 + 1)");
        assert_eq!(bd.iteration_stepsize, 10, "iteration_stepsize (stored 9 + 1)");

        // Word 5
        assert!(bd.tlast_suppress, "tlast_suppress");
        assert_eq!(bd.next_bd, 5, "next_bd");
        assert!(bd.use_next_bd, "use_next_bd");
        assert!(bd.valid, "valid");
        assert_eq!(bd.lock_rel_value, 3, "lock_rel_value");
        assert_eq!(bd.lock_rel_id, 2, "lock_rel_id");
        assert!(bd.lock_acq_enable, "lock_acq_enable");
        assert_eq!(bd.lock_acq_value, 1, "lock_acq_value");
        assert_eq!(bd.lock_acq_id, 7, "lock_acq_id");
    }

    /// Cross-validate regdb-driven parse_memtile() against AM025 bit positions.
    ///
    /// Constructs BD words where every field has a distinct, non-zero value
    /// placed at exact AM025 bit positions, then verifies all fields extract
    /// correctly through the regdb-based parser.
    #[test]
    fn test_memtile_bd_cross_validation() {
        // Exact bit positions from AM025 / aie_registers_aie2.json (memory_tile):
        //
        // DMA_BD0_0: Buffer_Length[16:0], Out_Of_Order_BD_ID[22:17],
        //            Packet_ID[27:23], Packet_Type[30:28], Enable_Packet[31]
        // DMA_BD0_1: Base_Address[18:0], Use_Next_BD[19], Next_BD[25:20],
        //            D0_Zero_Before[31:26]
        // DMA_BD0_2: D0_Stepsize[16:0], D0_Wrap[26:17], TLAST_Suppress[31]
        // DMA_BD0_3: D1_Stepsize[16:0], D1_Wrap[26:17], D1_Zero_Before[31:27]
        // DMA_BD0_4: D2_Stepsize[16:0], D2_Wrap[26:17], D2_Zero_Before[30:27],
        //            Enable_Compression[31]
        // DMA_BD0_5: D3_Stepsize[16:0], D0_Zero_After[22:17],
        //            D1_Zero_After[27:23], D2_Zero_After[31:28]
        // DMA_BD0_6: Iteration_Stepsize[16:0], Iteration_Wrap[22:17],
        //            Iteration_Current[28:23]
        // DMA_BD0_7: Lock_Acq_ID[7:0], Lock_Acq_Value[14:8],
        //            Lock_Acq_Enable[15], Lock_Rel_ID[23:16],
        //            Lock_Rel_Value[30:24], Valid_BD[31]

        let w0: u32 = (1 << 31) | (5 << 28) | (0x0A << 23) | (0x15 << 17) | 0x1234;
        let w1: u32 = (3 << 26) | (7 << 20) | (1 << 19) | 0x3_ABCD;
        let w2: u32 = (1 << 31) | (5 << 17) | 99; // tlast, d0_wrap=5, d0_step stored=99
        let w3: u32 = (2 << 27) | (8 << 17) | 49; // d1_zero_before=2, d1_wrap=8, d1_step=49
        let w4: u32 = (1 << 31) | (3 << 27) | (6 << 17) | 29; // compress, d2_zero_before=3, d2_wrap=6, d2_step=29
        let w5: u32 = (4 << 28) | (5 << 23) | (6 << 17) | 19; // d2_zero_after=4, d1_zero_after=5, d0_zero_after=6, d3_step=19
        let w6: u32 = (10 << 23) | (3 << 17) | 39; // iter_cur=10, iter_wrap stored=3, iter_step stored=39
        let w7: u32 = (1u32 << 31) | (3 << 24) | (0x42 << 16) | (1 << 15) | (2 << 8) | 0x37;

        let words = [w0, w1, w2, w3, w4, w5, w6, w7];
        let bd = BufferDescriptor::from_registers(&words, TileKind::Mem);

        // Word 0
        assert!(bd.enable_packet, "enable_packet");
        assert_eq!(bd.packet_type, 5, "packet_type");
        assert_eq!(bd.packet_id, 0x0A, "packet_id");
        assert_eq!(bd.ooo_bd_id, 0x15, "ooo_bd_id");
        assert_eq!(bd.length_words, 0x1234, "length_words");

        // Word 1
        assert_eq!(bd.d0_zero_before, 3, "d0_zero_before");
        assert_eq!(bd.next_bd, 7, "next_bd");
        assert!(bd.use_next_bd, "use_next_bd");
        assert_eq!(bd.base_addr_words, 0x3_ABCD, "base_addr_words");

        // Word 2 (stepsizes stored as actual-1)
        assert!(bd.tlast_suppress, "tlast_suppress");
        assert_eq!(bd.d0_wrap, 5, "d0_wrap");
        assert_eq!(bd.d0_stepsize, 100, "d0_stepsize (stored 99 + 1)");

        // Word 3
        assert_eq!(bd.d1_zero_before, 2, "d1_zero_before");
        assert_eq!(bd.d1_wrap, 8, "d1_wrap");
        assert_eq!(bd.d1_stepsize, 50, "d1_stepsize (stored 49 + 1)");

        // Word 4
        assert!(bd.compression_enable, "compression_enable");
        assert_eq!(bd.d2_zero_before, 3, "d2_zero_before");
        assert_eq!(bd.d2_wrap, 6, "d2_wrap");
        assert_eq!(bd.d2_stepsize, 30, "d2_stepsize (stored 29 + 1)");

        // Word 5
        assert_eq!(bd.d2_zero_after, 4, "d2_zero_after");
        assert_eq!(bd.d1_zero_after, 5, "d1_zero_after");
        assert_eq!(bd.d0_zero_after, 6, "d0_zero_after");
        assert_eq!(bd.d3_stepsize, 20, "d3_stepsize (stored 19 + 1)");

        // Word 6
        assert_eq!(bd.iteration_current, 10, "iteration_current");
        assert_eq!(bd.iteration_wrap, 4, "iteration_wrap (stored 3 + 1)");
        assert_eq!(bd.iteration_stepsize, 40, "iteration_stepsize (stored 39 + 1)");

        // Word 7
        assert!(bd.valid, "valid");
        assert_eq!(bd.lock_rel_value, 3, "lock_rel_value");
        assert_eq!(bd.lock_rel_id, 0x42, "lock_rel_id");
        assert!(bd.lock_acq_enable, "lock_acq_enable");
        assert_eq!(bd.lock_acq_value, 2, "lock_acq_value");
        assert_eq!(bd.lock_acq_id, 0x37, "lock_acq_id");
    }

    /// Test to_bd_config() conversion with a fully-populated BufferDescriptor.
    ///
    /// Constructs a BufferDescriptor directly (not via register parsing) to
    /// verify the unit conversions independently of bit layouts.
    #[test]
    fn test_to_bd_config_full() {
        let bd = BufferDescriptor {
            valid: true,
            base_addr_words: 0x1000,
            length_words: 256,
            d0_stepsize: 4, // actual words
            d0_wrap: 16,
            d1_stepsize: 8,
            d1_wrap: 4,
            d2_stepsize: 16,
            d2_wrap: 2,
            d3_stepsize: 32,        // MemTile only
            iteration_stepsize: 10, // actual words
            iteration_wrap: 3,      // actual count
            iteration_current: 0,
            lock_acq_enable: true,
            lock_acq_id: 5,
            lock_acq_value: 1,
            lock_rel_id: 5,
            lock_rel_value: -1,
            use_next_bd: true,
            next_bd: 3,
            enable_packet: true,
            packet_id: 7,
            packet_type: 2,
            ooo_bd_id: 10,
            tlast_suppress: true,
            compression_enable: true,
            ..Default::default()
        };

        let cfg = bd.to_bd_config();

        // Address and length: word units * 4 = byte units
        assert_eq!(cfg.base_addr, 0x1000 * 4);
        assert_eq!(cfg.length, 256 * 4);

        // Dimension strides: word strides * 4 = byte strides
        assert_eq!(cfg.d0.size, 16);
        assert_eq!(cfg.d0.stride, 4 * 4);
        assert_eq!(cfg.d1.size, 4);
        assert_eq!(cfg.d1.stride, 8 * 4);
        assert_eq!(cfg.d2.size, 2);
        assert_eq!(cfg.d2.stride, 16 * 4);
        // D3 size is implicit: length_words / (d0 * d1 * d2) = 256 / 128 = 2.
        assert_eq!(cfg.d3.size, 2);
        assert_eq!(cfg.d3.stride, 32 * 4);

        // Iteration: actual values -> stored-1 convention
        assert_eq!(cfg.iteration.wrap, 2, "wrap: actual 3 -> stored 2");
        assert_eq!(cfg.iteration.stepsize, 9, "stepsize: actual 10 -> stored 9");
        assert_eq!(cfg.iteration.current, 0);

        // Locks
        assert_eq!(cfg.acquire_lock, Some(5));
        assert_eq!(cfg.acquire_value, 1);
        assert_eq!(cfg.release_lock, Some(5));
        assert_eq!(cfg.release_value, -1);

        // Chaining
        assert_eq!(cfg.next_bd, Some(3));

        // Packet/compression
        assert!(cfg.enable_packet);
        assert_eq!(cfg.packet_id, 7);
        assert_eq!(cfg.packet_type, 2);
        assert_eq!(cfg.out_of_order_bd_id, 10);
        assert!(cfg.tlast_suppress);
        assert!(cfg.compression_enable);
        assert!(cfg.valid);
    }

    /// Test to_bd_config() with lock and chaining disabled.
    #[test]
    fn test_to_bd_config_no_locks_no_chain() {
        let bd = BufferDescriptor {
            valid: true,
            length_words: 64,
            lock_acq_enable: false,
            lock_acq_id: 3,    // Should be ignored
            lock_rel_value: 0, // 0 means no release
            lock_rel_id: 3,    // Should be ignored
            use_next_bd: false,
            next_bd: 5, // Should be ignored
            ..Default::default()
        };

        let cfg = bd.to_bd_config();

        assert_eq!(cfg.acquire_lock, None);
        assert_eq!(cfg.release_lock, None);
        assert_eq!(cfg.next_bd, None);
    }

    /// Test to_bd_config() with d0_wrap=0 (simple 1D contiguous transfer).
    #[test]
    fn test_to_bd_config_simple_1d() {
        let bd = BufferDescriptor {
            valid: true,
            length_words: 128,
            d0_stepsize: 1, // 1 word stride
            d0_wrap: 0,     // 0 means simple contiguous
            ..Default::default()
        };

        let cfg = bd.to_bd_config();

        // d0_wrap=0 -> d0.size should be length_words (128)
        assert_eq!(cfg.d0.size, 128);
        // d0_stepsize=1 word -> stride = 4 bytes
        assert_eq!(cfg.d0.stride, 4);
    }

    /// Round-trip test: parse shim BD from registers, convert to BdConfig,
    /// and verify all fields match expectations.
    #[test]
    fn test_to_bd_config_shim_round_trip() {
        // Use the existing shim BD parsing test values
        let words: [u32; 8] = [
            0x0000_1000, // BD_0: length=0x1000 words
            0x0000_0400, // BD_1: addr_low=0x100
            0x0000_0000, // BD_2: addr_high=0
            0x0010_0003, // BD_3: d0_wrap=1, d0_step stored=3 (actual=4)
            0x0000_0000, // BD_4
            0x0000_0000, // BD_5
            0x0000_0000, // BD_6
            0x0200_0000, // BD_7: valid=1
        ];

        let bd = BufferDescriptor::from_registers(&words, TileKind::ShimNoc);
        let cfg = bd.to_bd_config();

        assert!(cfg.valid);
        assert_eq!(cfg.base_addr, 0x100 * 4);
        assert_eq!(cfg.length, 0x1000 * 4);
        assert_eq!(cfg.d0.size, 1);
        assert_eq!(cfg.d0.stride, 4 * 4);
    }

    /// Shim BD: D2 wrap is derived from buffer_length / (d0_size * d1_size)
    /// since the shim BD register has no D2_Wrap field. This tests that a
    /// 4D transfer [1,4,64,64] with strides [0,4096,64,1] produces the
    /// correct d2_wrap=4 on a shim tile.
    #[test]
    fn test_shim_bd_implicit_d2_wrap() {
        // Simulates the BD for: dma_memcpy_nd(%arg0[0,0,0,0][1,4,64,64][0,4096,64,1])
        // In word units: D0=16 words (stride 1), D1=64 (stride 16), D2 stride=1024
        // Buffer_Length = 16*64*4 = 4096 words = 16384 bytes
        let words: [u32; 8] = [
            4096,        // BD_0: length=4096 words (16384 bytes)
            0x0000_0400, // BD_1: addr_low
            0x0000_0000, // BD_2: no packet
            // BD_3: d0_wrap=16, d0_step stored=0 (actual=1)
            (16 << 20) | 0,
            // BD_4: d1_wrap=64, d1_step stored=15 (actual=16)
            (64 << 20) | 15,
            // BD_5: d2_step stored=1023 (actual=1024)
            1023,
            // BD_6: no iteration
            0,
            // BD_7: valid=1
            1u32 << 25,
        ];

        let bd = BufferDescriptor::from_registers(&words, TileKind::ShimNoc);

        // Verify D2 wrap was derived from buffer_length / (d0 * d1)
        assert_eq!(bd.d2_wrap, 4, "d2_wrap should be 4096/(16*64)=4");
        assert_eq!(bd.d0_wrap, 16);
        assert_eq!(bd.d1_wrap, 64);
        assert_eq!(bd.d2_stepsize, 1024);

        let cfg = bd.to_bd_config();
        assert_eq!(cfg.d2.size, 4);
        assert_eq!(cfg.d2.stride, 1024 * 4); // 4096 bytes
    }

    /// Shim BD: simple 1D transfer should NOT produce a D2 wrap.
    #[test]
    fn test_shim_bd_simple_no_d2_wrap() {
        let words: [u32; 8] = [
            64,         // BD_0: length=64 words
            0,          // BD_1
            0,          // BD_2
            0,          // BD_3: d0_wrap=0 (simple), d0_step=0
            0,          // BD_4: d1_wrap=0, d1_step=0
            0,          // BD_5: d2_step=0
            0,          // BD_6
            1u32 << 25, // BD_7: valid=1
        ];

        let bd = BufferDescriptor::from_registers(&words, TileKind::ShimNoc);
        assert_eq!(bd.d2_wrap, 0, "simple 1D should have d2_wrap=0");
    }

    /// MemTile BD: D3 size is implicit, derived from
    /// buffer_length / (d0_size * d1_size * d2_size).
    ///
    /// Per aie-rt `AieMlMemTileMultiDimProp`, MemTile BDs support 4
    /// explicit dimensions, but D3 has only a StepSize register --
    /// its size is implicit. This mirrors the derivation used by
    /// compute/shim BDs for d2_wrap (which also lack a register
    /// field for their outermost dim size).
    ///
    /// Reproduces the dma_complex_dims failure mode: a 4D objectfifo
    /// [<size=4,stride=40>, <size=1,stride=5>, <size=8,stride=5>,
    /// <size=5,stride=1>] must fully tile 160 i32 words with no
    /// underrun.
    #[test]
    fn test_memtile_bd_implicit_d3_size() {
        // Simulates mem_to_comp producer BD for dma_complex_dims:
        //   MLIR dims (outermost->innermost): 4/40, 1/5, 8/5, 5/1
        //   Hardware mapping:
        //     D0: wrap=5, step stored=0 (actual=1 word = 4 bytes)
        //     D1: wrap=8, step stored=4 (actual=5 words = 20 bytes)
        //     D2: wrap=1, step stored=4 (actual=5 words = 20 bytes)
        //     D3: step stored=39 (actual=40 words = 160 bytes), size implicit
        //     length=160 words, addr=0xA0000/4=0x28000
        //
        // AM025 MemTile BD bit positions:
        //   BD_0: Buffer_Length[16:0]
        //   BD_1: Base_Address[18:0]
        //   BD_2: D0_Stepsize[16:0], D0_Wrap[26:17]
        //   BD_3: D1_Stepsize[16:0], D1_Wrap[26:17]
        //   BD_4: D2_Stepsize[16:0], D2_Wrap[26:17]
        //   BD_5: D3_Stepsize[16:0]     (no D3_Wrap -- size is implicit)
        //   BD_7: Valid_BD[31]
        let words: [u32; 8] = [
            160,              // BD_0: buffer_length = 160 words
            0x2_8000,         // BD_1: base_address = 0x28000
            (5u32 << 17) | 0, // BD_2: d0_wrap=5, d0_step stored=0
            (8u32 << 17) | 4, // BD_3: d1_wrap=8, d1_step stored=4
            (1u32 << 17) | 4, // BD_4: d2_wrap=1, d2_step stored=4
            39,               // BD_5: d3_step stored=39 (actual=40)
            0,                // BD_6: no iteration
            1u32 << 31,       // BD_7: valid=1
        ];

        let bd = BufferDescriptor::from_registers(&words, TileKind::Mem);
        assert_eq!(bd.d0_wrap, 5);
        assert_eq!(bd.d1_wrap, 8);
        assert_eq!(bd.d2_wrap, 1);
        assert_eq!(bd.d0_stepsize, 1);
        assert_eq!(bd.d1_stepsize, 5);
        assert_eq!(bd.d2_stepsize, 5);
        assert_eq!(bd.d3_stepsize, 40);
        assert_eq!(bd.length_words, 160);

        let cfg = bd.to_bd_config();
        // D3 size must be derived: 160 / (5 * 8 * 1) = 4.
        assert_eq!(cfg.d3.size, 4, "d3 size should be 160/(5*8*1)=4");
        assert_eq!(cfg.d3.stride, 40 * 4, "d3 stride in bytes");

        // Total elements across all 4 dims must match length_words so the
        // address generator produces all 160 addresses, not 40.
        let total: u32 = cfg.d0.effective_size()
            * cfg.d1.effective_size()
            * cfg.d2.effective_size()
            * cfg.d3.effective_size();
        assert_eq!(total, cfg.length / 4, "dim product must match length/4");
    }

    /// MemTile BD: for simple 1D transfers where length_words equals
    /// d0*d1*d2, d3.size stays 0 (effective 1) so the dim product
    /// matches length exactly without spurious d3 iterations.
    #[test]
    fn test_memtile_bd_simple_1d_no_d3() {
        // 1D contiguous BD: d0_wrap=64, others unused.
        // D0_Wrap field at [26:17] per AM025.
        let words: [u32; 8] = [
            64,                // length=64 words
            0,                 // addr=0
            (64u32 << 17) | 0, // d0_wrap=64, d0_step stored=0
            0,
            0,
            0, // d3_step stored=0 (actual=1 after +1)
            0,
            1u32 << 31, // valid
        ];

        // Note: parse_memtile stores d3_stepsize = extract + 1. With
        // stored=0, actual d3_stepsize=1, which is still > 0. This
        // matches the reset value; we want to verify that the
        // derivation doesn't inflate d3 spuriously for simple
        // transfers.
        let bd = BufferDescriptor::from_registers(&words, TileKind::Mem);
        assert_eq!(bd.d3_stepsize, 1, "d3_stepsize reset value after +1");

        let cfg = bd.to_bd_config();
        // d0=64, d1=0 (eff 1), d2=0 (eff 1). product=64. length_words=64.
        // length_words > product is FALSE (equal), so d3_size stays 0.
        assert_eq!(cfg.d3.size, 0, "no d3 for simple 1D transfer");
        let total: u32 = cfg.d0.effective_size()
            * cfg.d1.effective_size()
            * cfg.d2.effective_size()
            * cfg.d3.effective_size();
        assert_eq!(total, cfg.length / 4);
    }

    #[test]
    fn test_bd_address_conversion() {
        assert_eq!(BufferDescriptor::byte_to_word_addr(0), 0);
        assert_eq!(BufferDescriptor::byte_to_word_addr(4), 1);
        assert_eq!(BufferDescriptor::byte_to_word_addr(1024), 256);

        assert_eq!(BufferDescriptor::word_to_byte_addr(0), 0);
        assert_eq!(BufferDescriptor::word_to_byte_addr(1), 4);
        assert_eq!(BufferDescriptor::word_to_byte_addr(256), 1024);
    }
}
