//! DMA FIFO mode and FIFO data structures.
//!
//! # AIE2 DMA FIFO Mode Status
//!
//! The AIE1 "DMA FIFO mode" (BD-level pairing of S2MM and MM2S channels through
//! a shared circular buffer with hardware-managed read/write pointers) is
//! **not available on AIE2 (AIE-ML)**. Per aie-rt `xaiemlgbl_reginit.c`:
//!
//! ```text
//! .FifoMode = XAIE_FEATURE_UNAVAILABLE,  // MemTile, Tile, and Shim DMA
//! ```
//!
//! The aie-rt API `XAie_DmaConfigFifoMode()` returns `XAIE_FEATURE_NOT_SUPPORTED`
//! when called on AIE-ML devices. The BD descriptor's `FifoMode` field and the
//! `XAie_DmaFifoCounter` enum (`NONE=0, COUNTER_0=2, COUNTER_1=3`) are AIE1-only.
//!
//! # What this module provides
//!
//! 1. **`DmaFifo`** -- A circular buffer suitable for DMA FIFO mode emulation
//!    (AIE1 future support) and for any internal buffering needs. Operates at
//!    byte granularity with word-aligned capacity.
//!
//! 2. **`FotCountFifo`** -- The FoT (Finish-on-TLAST) Count FIFO that IS present
//!    in AIE2. Each S2MM channel has a read-only pop register
//!    (`DMA_S2MM_FoT_Count_FIFO_Pop_N`) that reports per-transfer word counts
//!    when the channel operates in `FoT_counts_from_mm_register` mode. Per AM025:
//!    - Bit 31: Valid (0 = not in FoT mode or FIFO empty)
//!    - Bit 30: Last_in_Task
//!    - Bits 29:24: BD_ID
//!    - Bits 17:0: Write_Count (32-bit words written)
//!
//! # Hardware derivation
//!
//! - DMA FIFO mode availability: `aie-rt/driver/src/global/xaiemlgbl_reginit.c`
//!   lines 428, 665, 917 (all `XAIE_FEATURE_UNAVAILABLE` for AIE-ML)
//! - FoT Count FIFO register layout: AM025 register database
//!   (`aie_registers_aie2.json`, registers `DMA_S2MM_FoT_Count_FIFO_Pop_0..1`)
//! - DMA FIFO counter enum: `aie-rt/driver/src/dma/xaie_dma.h` lines 38-41
//! - FIFO mode config API: `aie-rt/driver/src/dma/xaie_dma.c` lines 574-597

use std::collections::VecDeque;

/// DMA data width in bytes (32-bit words).
const WORD_SIZE: u32 = 4;

// ---------------------------------------------------------------------------
// DMA FIFO circular buffer
// ---------------------------------------------------------------------------

/// Circular buffer for DMA FIFO mode.
///
/// Manages a byte-level circular buffer with word-aligned depth. On AIE1,
/// hardware pairs an S2MM channel (writer) with an MM2S channel (reader)
/// through a shared memory region. The FIFO counter tracks fill level and
/// provides backpressure: S2MM stalls when full, MM2S stalls when empty.
///
/// This implementation is architecture-neutral -- it provides the circular
/// buffer mechanics without assuming a specific tile type. The caller
/// (DMA engine) is responsible for enforcing feature availability
/// (`XAIE_FEATURE_UNAVAILABLE` on AIE2).
///
/// # Register interface
///
/// The FIFO exposes a minimal register interface for CDO configuration:
///
/// | Offset | Name          | Access | Description                    |
/// |--------|---------------|--------|--------------------------------|
/// | 0x00   | CTRL          | R/W    | Bit 0: enable                  |
/// | 0x04   | BASE_ADDR     | R/W    | Base address in local memory   |
/// | 0x08   | DEPTH         | R/W    | Depth in 32-bit words          |
/// | 0x0C   | COUNT         | RO     | Current fill level (words)     |
///
/// These offsets are internal to this module. The parent DMA engine maps
/// them to actual MMIO offsets as appropriate for the tile type.
#[derive(Debug, Clone)]
pub struct DmaFifo {
    /// Whether FIFO mode is enabled.
    enabled: bool,
    /// Base address in local memory (byte address).
    base_addr: u32,
    /// FIFO depth in 32-bit words.
    depth_words: u32,
    /// Write pointer (word offset from base, wraps at depth).
    write_pointer: u32,
    /// Read pointer (word offset from base, wraps at depth).
    read_pointer: u32,
    /// Current fill level in 32-bit words.
    count: u32,
}

// Register offsets within the FIFO register block.
/// Control register: bit 0 = enable.
pub const REG_CTRL: u32 = 0x00;
/// Base address register (word address).
pub const REG_BASE_ADDR: u32 = 0x04;
/// Depth register (word count).
pub const REG_DEPTH: u32 = 0x08;
/// Current fill-level register (read-only, word count).
pub const REG_COUNT: u32 = 0x0C;
/// Total register space consumed by the FIFO block.
pub const REG_BLOCK_SIZE: u32 = 0x10;

impl DmaFifo {
    /// Create a new disabled, unconfigured FIFO.
    pub fn new() -> Self {
        Self {
            enabled: false,
            base_addr: 0,
            depth_words: 0,
            write_pointer: 0,
            read_pointer: 0,
            count: 0,
        }
    }

    /// Configure the FIFO base address and depth.
    ///
    /// `base_addr` is the byte address of the FIFO region in local memory.
    /// `depth` is the capacity in 32-bit words.
    ///
    /// Resets the FIFO pointers so that configuration always starts clean.
    pub fn configure(&mut self, base_addr: u32, depth: u32) {
        self.base_addr = base_addr;
        self.depth_words = depth;
        self.write_pointer = 0;
        self.read_pointer = 0;
        self.count = 0;
    }

    /// Enable the FIFO.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable the FIFO. Does not reset pointers -- call [`reset`] explicitly
    /// if you need to drain state.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Returns `true` if the FIFO is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Write data into the FIFO circular buffer.
    ///
    /// Returns the number of bytes actually written. May be less than
    /// `data.len()` if the FIFO becomes full (backpressure). Data is
    /// consumed in 32-bit word granularity -- partial trailing words in
    /// `data` are silently truncated.
    ///
    /// Returns 0 if the FIFO is disabled, full, or has zero depth.
    pub fn write(&mut self, data: &[u8]) -> usize {
        if !self.enabled || self.depth_words == 0 {
            return 0;
        }

        let available = self.depth_words - self.count;
        let words_in_data = (data.len() / WORD_SIZE as usize) as u32;
        let words_to_write = words_in_data.min(available);

        // We don't actually store data here -- the real DMA FIFO uses
        // local tile memory as its backing store. This struct just tracks
        // the pointers and fill level. The caller writes data to
        // base_addr + write_pointer * 4 in the tile's memory array.
        for _ in 0..words_to_write {
            self.write_pointer = (self.write_pointer + 1) % self.depth_words;
            self.count += 1;
        }

        (words_to_write * WORD_SIZE) as usize
    }

    /// Read data from the FIFO circular buffer.
    ///
    /// Returns the number of bytes actually read into `buf`. May be less
    /// than `buf.len()` if the FIFO becomes empty. Operates in 32-bit
    /// word granularity.
    ///
    /// Returns 0 if the FIFO is disabled, empty, or has zero depth.
    pub fn read(&mut self, buf: &mut [u8]) -> usize {
        if !self.enabled || self.depth_words == 0 {
            return 0;
        }

        let words_in_buf = (buf.len() / WORD_SIZE as usize) as u32;
        let words_to_read = words_in_buf.min(self.count);

        for _ in 0..words_to_read {
            self.read_pointer = (self.read_pointer + 1) % self.depth_words;
            self.count -= 1;
        }

        (words_to_read * WORD_SIZE) as usize
    }

    /// Returns `true` if the FIFO has no data.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Returns `true` if the FIFO is completely full.
    /// An enabled zero-depth FIFO is vacuously full (cannot accept data).
    /// A disabled/unconfigured FIFO is not full.
    pub fn is_full(&self) -> bool {
        self.enabled && self.count == self.depth_words
    }

    /// Current fill level in 32-bit words.
    pub fn fill_level(&self) -> u32 {
        self.count
    }

    /// Total capacity in 32-bit words.
    pub fn capacity(&self) -> u32 {
        self.depth_words
    }

    /// Base address (byte address in local memory).
    pub fn base_addr(&self) -> u32 {
        self.base_addr
    }

    /// Current write pointer (word offset from base, wraps at depth).
    pub fn write_pointer(&self) -> u32 {
        self.write_pointer
    }

    /// Current read pointer (word offset from base, wraps at depth).
    pub fn read_pointer(&self) -> u32 {
        self.read_pointer
    }

    /// Reset the FIFO to empty state without changing configuration.
    pub fn reset(&mut self) {
        self.write_pointer = 0;
        self.read_pointer = 0;
        self.count = 0;
    }

    /// Read a FIFO control register.
    ///
    /// Returns `Some(value)` for valid offsets, `None` for unrecognized ones.
    /// See module-level register offset constants for the layout.
    pub fn read_register(&self, offset: u32) -> Option<u32> {
        match offset {
            REG_CTRL => Some(if self.enabled { 1 } else { 0 }),
            REG_BASE_ADDR => Some(self.base_addr),
            REG_DEPTH => Some(self.depth_words),
            REG_COUNT => Some(self.count),
            _ => None,
        }
    }

    /// Write a FIFO control register.
    ///
    /// Returns `true` if the offset was valid and the write was applied,
    /// `false` for unrecognized or read-only offsets.
    pub fn write_register(&mut self, offset: u32, value: u32) -> bool {
        match offset {
            REG_CTRL => {
                if value & 1 != 0 {
                    self.enable();
                } else {
                    self.disable();
                }
                true
            }
            REG_BASE_ADDR => {
                self.base_addr = value;
                true
            }
            REG_DEPTH => {
                self.depth_words = value;
                // Reset pointers when depth changes to avoid stale state.
                self.reset();
                true
            }
            // REG_COUNT is read-only.
            REG_COUNT => false,
            _ => false,
        }
    }
}

impl Default for DmaFifo {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FoT Count FIFO (AIE2)
// ---------------------------------------------------------------------------

/// Entry in the FoT (Finish-on-TLAST) Count FIFO.
///
/// When an S2MM channel operates in `FoT_counts_from_mm_register` mode,
/// the hardware records one entry per completed transfer into this FIFO.
/// The core reads `DMA_S2MM_FoT_Count_FIFO_Pop_N` to pop entries. Per AM025:
///
/// - Bit 31: Valid (1 = entry is valid, 0 = FIFO empty or wrong mode)
/// - Bit 30: Last_in_Task (1 = last transfer in the current task)
/// - Bits 29:24: BD_ID (which BD completed)
/// - Bits 17:0: Write_Count (32-bit words written to memory)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FotCountEntry {
    /// Which BD completed this transfer.
    pub bd_id: u8,
    /// Number of 32-bit words written to memory.
    pub write_count: u32,
    /// Whether this was the last transfer in the task.
    pub last_in_task: bool,
}

impl FotCountEntry {
    /// Encode this entry as the 32-bit register value with Valid=1.
    pub fn to_register(&self) -> u32 {
        let valid = 1u32 << 31;
        let last = if self.last_in_task { 1u32 << 30 } else { 0 };
        let bd = ((self.bd_id as u32) & 0x3F) << 24;
        let count = (self.write_count) & 0x3FFFF;
        valid | last | bd | count
    }
}

/// FoT Count FIFO for a single S2MM channel.
///
/// The hardware FIFO depth is not specified in AM025 beyond the "count FIFO
/// full" error bit. We use a reasonable depth (16 entries) matching the
/// maximum BD count per compute tile.
///
/// Reading the pop register returns the front entry (with Valid=1) and
/// removes it. Reading when empty returns 0 (Valid=0).
#[derive(Debug, Clone)]
pub struct FotCountFifo {
    entries: VecDeque<FotCountEntry>,
    /// Maximum entries before the "Count_FIFO_Full" error fires.
    capacity: usize,
}

/// Default FoT Count FIFO capacity.
///
/// AM025 does not specify the exact depth, but the error bit
/// `Stalled_TCT_or_Count_FIFO_Full` implies a fixed hardware depth.
/// 16 matches the BD count for compute tiles.
const FOT_COUNT_FIFO_DEFAULT_CAPACITY: usize = 16;

impl FotCountFifo {
    /// Create a new empty FoT Count FIFO with default capacity.
    pub fn new() -> Self {
        Self {
            entries: VecDeque::with_capacity(FOT_COUNT_FIFO_DEFAULT_CAPACITY),
            capacity: FOT_COUNT_FIFO_DEFAULT_CAPACITY,
        }
    }

    /// Create a FoT Count FIFO with a specific capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Push a completed transfer entry. Returns `true` if accepted,
    /// `false` if the FIFO is full (mirrors the `Count_FIFO_Full` error).
    pub fn push(&mut self, entry: FotCountEntry) -> bool {
        if self.entries.len() >= self.capacity {
            return false;
        }
        self.entries.push_back(entry);
        true
    }

    /// Pop the front entry, returning the 32-bit register value.
    ///
    /// Returns 0 (Valid=0) if the FIFO is empty, matching the hardware
    /// behavior described in AM025: "0: channel not in
    /// FoT_counts_from_mm_register mode OR FIFO is empty."
    pub fn pop_register(&mut self) -> u32 {
        match self.entries.pop_front() {
            Some(entry) => entry.to_register(),
            None => 0, // Valid=0
        }
    }

    /// Peek at the front entry without removing it.
    pub fn peek(&self) -> Option<&FotCountEntry> {
        self.entries.front()
    }

    /// Number of entries currently in the FIFO.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the FIFO is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether the FIFO is full.
    pub fn is_full(&self) -> bool {
        self.entries.len() >= self.capacity
    }

    /// Maximum capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

impl Default for FotCountFifo {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // DmaFifo tests
    // -----------------------------------------------------------------------

    #[test]
    fn new_fifo_is_disabled_and_empty() {
        let fifo = DmaFifo::new();
        assert!(!fifo.is_enabled());
        assert!(fifo.is_empty());
        assert!(!fifo.is_full());
        assert_eq!(fifo.fill_level(), 0);
        assert_eq!(fifo.capacity(), 0);
    }

    #[test]
    fn configure_sets_base_and_depth() {
        let mut fifo = DmaFifo::new();
        fifo.configure(0x1000, 64);
        assert_eq!(fifo.base_addr(), 0x1000);
        assert_eq!(fifo.capacity(), 64);
        assert!(fifo.is_empty());
    }

    #[test]
    fn enable_disable() {
        let mut fifo = DmaFifo::new();
        fifo.configure(0, 16);
        assert!(!fifo.is_enabled());

        fifo.enable();
        assert!(fifo.is_enabled());

        fifo.disable();
        assert!(!fifo.is_enabled());
    }

    #[test]
    fn write_then_read_returns_same_byte_count() {
        let mut fifo = DmaFifo::new();
        fifo.configure(0x2000, 8); // 8 words = 32 bytes
        fifo.enable();

        // Write 3 words (12 bytes)
        let data = [0u8; 12];
        let written = fifo.write(&data);
        assert_eq!(written, 12);
        assert_eq!(fifo.fill_level(), 3);

        // Read back 3 words
        let mut buf = [0u8; 12];
        let read = fifo.read(&mut buf);
        assert_eq!(read, 12);
        assert_eq!(fifo.fill_level(), 0);
        assert!(fifo.is_empty());
    }

    #[test]
    fn circular_buffer_wraps_correctly() {
        let mut fifo = DmaFifo::new();
        fifo.configure(0, 4); // 4-word capacity
        fifo.enable();

        // Fill to capacity
        let data = [0u8; 16]; // 4 words
        let written = fifo.write(&data);
        assert_eq!(written, 16);
        assert!(fifo.is_full());
        assert_eq!(fifo.write_pointer(), 0); // wrapped around

        // Read 2 words -- frees space
        let mut buf = [0u8; 8];
        let read = fifo.read(&mut buf);
        assert_eq!(read, 8);
        assert_eq!(fifo.fill_level(), 2);
        assert_eq!(fifo.read_pointer(), 2);

        // Write 2 more words -- write pointer wraps
        let data2 = [0u8; 8];
        let written2 = fifo.write(&data2);
        assert_eq!(written2, 8);
        assert!(fifo.is_full());
        assert_eq!(fifo.write_pointer(), 2); // 0 + 2 = 2

        // Read remaining 4 words
        let mut buf2 = [0u8; 16];
        let read2 = fifo.read(&mut buf2);
        assert_eq!(read2, 16);
        assert!(fifo.is_empty());
        assert_eq!(fifo.read_pointer(), 2); // wrapped: (2 + 4) % 4 = 2
    }

    #[test]
    fn full_prevents_further_writes() {
        let mut fifo = DmaFifo::new();
        fifo.configure(0, 2); // 2 words
        fifo.enable();

        let data = [0u8; 8]; // exactly 2 words
        assert_eq!(fifo.write(&data), 8);
        assert!(fifo.is_full());

        // Attempt to write more
        let more = [0u8; 4];
        assert_eq!(fifo.write(&more), 0);
        assert!(fifo.is_full());
        assert_eq!(fifo.fill_level(), 2);
    }

    #[test]
    fn empty_prevents_reads() {
        let mut fifo = DmaFifo::new();
        fifo.configure(0, 8);
        fifo.enable();

        let mut buf = [0u8; 16];
        assert_eq!(fifo.read(&mut buf), 0);
        assert!(fifo.is_empty());
    }

    #[test]
    fn fill_level_tracks_correctly() {
        let mut fifo = DmaFifo::new();
        fifo.configure(0, 10);
        fifo.enable();

        // Write 3 words
        assert_eq!(fifo.write(&[0u8; 12]), 12);
        assert_eq!(fifo.fill_level(), 3);

        // Write 5 more words
        assert_eq!(fifo.write(&[0u8; 20]), 20);
        assert_eq!(fifo.fill_level(), 8);

        // Read 4 words
        let mut buf = [0u8; 16];
        assert_eq!(fifo.read(&mut buf), 16);
        assert_eq!(fifo.fill_level(), 4);

        // Read 4 more
        assert_eq!(fifo.read(&mut buf), 16);
        assert_eq!(fifo.fill_level(), 0);
    }

    #[test]
    fn partial_write_when_near_full() {
        let mut fifo = DmaFifo::new();
        fifo.configure(0, 4);
        fifo.enable();

        // Fill 3 of 4 words
        assert_eq!(fifo.write(&[0u8; 12]), 12);
        assert_eq!(fifo.fill_level(), 3);
        assert!(!fifo.is_full());

        // Try to write 3 words, only 1 fits
        assert_eq!(fifo.write(&[0u8; 12]), 4); // 1 word = 4 bytes
        assert!(fifo.is_full());
    }

    #[test]
    fn partial_read_when_near_empty() {
        let mut fifo = DmaFifo::new();
        fifo.configure(0, 8);
        fifo.enable();

        // Write 2 words
        assert_eq!(fifo.write(&[0u8; 8]), 8);
        assert_eq!(fifo.fill_level(), 2);

        // Try to read 5 words, only 2 available
        let mut buf = [0u8; 20];
        assert_eq!(fifo.read(&mut buf), 8); // 2 words = 8 bytes
        assert!(fifo.is_empty());
    }

    #[test]
    fn disabled_fifo_rejects_writes_and_reads() {
        let mut fifo = DmaFifo::new();
        fifo.configure(0, 8);
        // Don't enable

        assert_eq!(fifo.write(&[0u8; 8]), 0);
        let mut buf = [0u8; 8];
        assert_eq!(fifo.read(&mut buf), 0);
    }

    #[test]
    fn reset_clears_all_state() {
        let mut fifo = DmaFifo::new();
        fifo.configure(0x3000, 8);
        fifo.enable();

        // Fill partially
        fifo.write(&[0u8; 12]);
        assert_eq!(fifo.fill_level(), 3);
        assert_ne!(fifo.write_pointer(), 0);

        fifo.reset();
        assert!(fifo.is_empty());
        assert_eq!(fifo.write_pointer(), 0);
        assert_eq!(fifo.read_pointer(), 0);
        assert_eq!(fifo.fill_level(), 0);

        // Config preserved
        assert_eq!(fifo.base_addr(), 0x3000);
        assert_eq!(fifo.capacity(), 8);
        assert!(fifo.is_enabled()); // enable state preserved
    }

    #[test]
    fn register_read_ctrl() {
        let mut fifo = DmaFifo::new();
        assert_eq!(fifo.read_register(REG_CTRL), Some(0));
        fifo.enable();
        assert_eq!(fifo.read_register(REG_CTRL), Some(1));
    }

    #[test]
    fn register_read_base_addr() {
        let mut fifo = DmaFifo::new();
        fifo.configure(0x4000, 16);
        assert_eq!(fifo.read_register(REG_BASE_ADDR), Some(0x4000));
    }

    #[test]
    fn register_read_depth() {
        let mut fifo = DmaFifo::new();
        fifo.configure(0, 32);
        assert_eq!(fifo.read_register(REG_DEPTH), Some(32));
    }

    #[test]
    fn register_read_count() {
        let mut fifo = DmaFifo::new();
        fifo.configure(0, 8);
        fifo.enable();
        fifo.write(&[0u8; 12]); // 3 words
        assert_eq!(fifo.read_register(REG_COUNT), Some(3));
    }

    #[test]
    fn register_read_invalid_offset() {
        let fifo = DmaFifo::new();
        assert_eq!(fifo.read_register(0xFF), None);
    }

    #[test]
    fn register_write_ctrl() {
        let mut fifo = DmaFifo::new();
        assert!(fifo.write_register(REG_CTRL, 1));
        assert!(fifo.is_enabled());
        assert!(fifo.write_register(REG_CTRL, 0));
        assert!(!fifo.is_enabled());
    }

    #[test]
    fn register_write_base_addr() {
        let mut fifo = DmaFifo::new();
        assert!(fifo.write_register(REG_BASE_ADDR, 0x5000));
        assert_eq!(fifo.base_addr(), 0x5000);
    }

    #[test]
    fn register_write_depth_resets_pointers() {
        let mut fifo = DmaFifo::new();
        fifo.configure(0, 4);
        fifo.enable();
        fifo.write(&[0u8; 8]); // 2 words

        // Changing depth resets the FIFO
        assert!(fifo.write_register(REG_DEPTH, 16));
        assert_eq!(fifo.capacity(), 16);
        assert!(fifo.is_empty());
        assert_eq!(fifo.write_pointer(), 0);
    }

    #[test]
    fn register_write_count_is_readonly() {
        let mut fifo = DmaFifo::new();
        assert!(!fifo.write_register(REG_COUNT, 42));
    }

    #[test]
    fn register_write_invalid_offset() {
        let mut fifo = DmaFifo::new();
        assert!(!fifo.write_register(0xFF, 0));
    }

    #[test]
    fn register_roundtrip() {
        let mut fifo = DmaFifo::new();

        // Write config via registers
        fifo.write_register(REG_BASE_ADDR, 0x8000);
        fifo.write_register(REG_DEPTH, 64);
        fifo.write_register(REG_CTRL, 1);

        // Read back
        assert_eq!(fifo.read_register(REG_BASE_ADDR), Some(0x8000));
        assert_eq!(fifo.read_register(REG_DEPTH), Some(64));
        assert_eq!(fifo.read_register(REG_CTRL), Some(1));
    }

    #[test]
    fn steady_state_streaming() {
        // Simulates continuous producer/consumer operation.
        let mut fifo = DmaFifo::new();
        fifo.configure(0, 8); // 8-word buffer
        fifo.enable();

        let mut total_written = 0u32;
        let mut total_read = 0u32;

        // Run 20 cycles of alternating write/read, 2 words each.
        for _ in 0..20 {
            let w = fifo.write(&[0u8; 8]) as u32; // 2 words
            total_written += w / 4;

            let mut buf = [0u8; 8];
            let r = fifo.read(&mut buf) as u32; // 2 words
            total_read += r / 4;
        }

        // Should have moved all data through (write before read each cycle,
        // so the FIFO never overflows with depth=8 and 2-word bursts).
        assert_eq!(total_written, 40);
        assert_eq!(total_read, 40);
        assert!(fifo.is_empty());
    }

    #[test]
    fn zero_depth_fifo_rejects_everything() {
        let mut fifo = DmaFifo::new();
        fifo.configure(0, 0);
        fifo.enable();

        assert_eq!(fifo.write(&[0u8; 4]), 0);
        let mut buf = [0u8; 4];
        assert_eq!(fifo.read(&mut buf), 0);
        assert!(fifo.is_empty());
        // A zero-depth FIFO is vacuously both empty and full.
        assert!(fifo.is_full());
    }

    #[test]
    fn sub_word_data_truncated() {
        let mut fifo = DmaFifo::new();
        fifo.configure(0, 4);
        fifo.enable();

        // 7 bytes = 1 word + 3 trailing bytes (truncated)
        assert_eq!(fifo.write(&[0u8; 7]), 4);
        assert_eq!(fifo.fill_level(), 1);

        // 3 bytes in buffer = 0 complete words
        let mut buf = [0u8; 3];
        assert_eq!(fifo.read(&mut buf), 0);
        assert_eq!(fifo.fill_level(), 1);
    }

    #[test]
    fn backpressure_signaling() {
        let mut fifo = DmaFifo::new();
        fifo.configure(0, 2);
        fifo.enable();

        assert!(!fifo.is_full());
        assert!(fifo.is_empty());

        fifo.write(&[0u8; 4]); // 1 word
        assert!(!fifo.is_full());
        assert!(!fifo.is_empty());

        fifo.write(&[0u8; 4]); // 2 words -- now full
        assert!(fifo.is_full());
        assert!(!fifo.is_empty());

        let mut buf = [0u8; 4];
        fifo.read(&mut buf); // 1 word out
        assert!(!fifo.is_full());
        assert!(!fifo.is_empty());

        fifo.read(&mut buf); // empty again
        assert!(!fifo.is_full());
        assert!(fifo.is_empty());
    }

    #[test]
    fn default_matches_new() {
        let a = DmaFifo::new();
        let b = DmaFifo::default();
        assert_eq!(a.is_enabled(), b.is_enabled());
        assert_eq!(a.base_addr(), b.base_addr());
        assert_eq!(a.capacity(), b.capacity());
        assert_eq!(a.fill_level(), b.fill_level());
    }

    #[test]
    fn configure_resets_previous_state() {
        let mut fifo = DmaFifo::new();
        fifo.configure(0, 4);
        fifo.enable();
        fifo.write(&[0u8; 8]); // 2 words
        assert_eq!(fifo.fill_level(), 2);

        // Reconfigure -- should clear pointers
        fifo.configure(0x1000, 16);
        assert_eq!(fifo.fill_level(), 0);
        assert_eq!(fifo.base_addr(), 0x1000);
        assert_eq!(fifo.capacity(), 16);
        assert!(fifo.is_empty());
    }

    // -----------------------------------------------------------------------
    // FotCountFifo tests
    // -----------------------------------------------------------------------

    #[test]
    fn fot_entry_to_register_encoding() {
        let entry = FotCountEntry {
            bd_id: 5,
            write_count: 256,
            last_in_task: false,
        };
        let reg = entry.to_register();
        // Bit 31 = Valid = 1
        assert_ne!(reg & (1 << 31), 0);
        // Bit 30 = Last_in_Task = 0
        assert_eq!(reg & (1 << 30), 0);
        // Bits 29:24 = BD_ID = 5
        assert_eq!((reg >> 24) & 0x3F, 5);
        // Bits 17:0 = Write_Count = 256
        assert_eq!(reg & 0x3FFFF, 256);
    }

    #[test]
    fn fot_entry_last_in_task() {
        let entry = FotCountEntry {
            bd_id: 15,
            write_count: 1024,
            last_in_task: true,
        };
        let reg = entry.to_register();
        assert_ne!(reg & (1 << 31), 0); // Valid
        assert_ne!(reg & (1 << 30), 0); // Last_in_Task
        assert_eq!((reg >> 24) & 0x3F, 15);
        assert_eq!(reg & 0x3FFFF, 1024);
    }

    #[test]
    fn fot_entry_max_fields() {
        let entry = FotCountEntry {
            bd_id: 63,           // 6-bit max
            write_count: 0x3FFFF, // 18-bit max
            last_in_task: true,
        };
        let reg = entry.to_register();
        // Verify fields don't overflow into each other.
        assert_eq!((reg >> 24) & 0x3F, 63);
        assert_eq!(reg & 0x3FFFF, 0x3FFFF);
        assert_ne!(reg & (1 << 31), 0);
        assert_ne!(reg & (1 << 30), 0);
    }

    #[test]
    fn fot_fifo_empty_pop_returns_zero() {
        let mut fifo = FotCountFifo::new();
        assert_eq!(fifo.pop_register(), 0); // Valid=0
        assert!(fifo.is_empty());
    }

    #[test]
    fn fot_fifo_push_pop_roundtrip() {
        let mut fifo = FotCountFifo::new();
        let entry = FotCountEntry {
            bd_id: 3,
            write_count: 128,
            last_in_task: false,
        };
        assert!(fifo.push(entry));
        assert_eq!(fifo.len(), 1);
        assert!(!fifo.is_empty());

        let reg = fifo.pop_register();
        assert_ne!(reg & (1 << 31), 0); // Valid
        assert_eq!((reg >> 24) & 0x3F, 3);
        assert_eq!(reg & 0x3FFFF, 128);
        assert!(fifo.is_empty());
    }

    #[test]
    fn fot_fifo_ordering() {
        let mut fifo = FotCountFifo::new();
        for i in 0..4 {
            fifo.push(FotCountEntry {
                bd_id: i,
                write_count: (i as u32 + 1) * 100,
                last_in_task: i == 3,
            });
        }
        assert_eq!(fifo.len(), 4);

        for i in 0..4 {
            let reg = fifo.pop_register();
            assert_eq!((reg >> 24) & 0x3F, i as u32);
            assert_eq!(reg & 0x3FFFF, (i as u32 + 1) * 100);
        }
        assert!(fifo.is_empty());
    }

    #[test]
    fn fot_fifo_full_rejects() {
        let mut fifo = FotCountFifo::with_capacity(2);
        let entry = FotCountEntry { bd_id: 0, write_count: 10, last_in_task: false };

        assert!(fifo.push(entry));
        assert!(fifo.push(entry));
        assert!(fifo.is_full());
        assert!(!fifo.push(entry)); // rejected
        assert_eq!(fifo.len(), 2);
    }

    #[test]
    fn fot_fifo_peek() {
        let mut fifo = FotCountFifo::new();
        assert!(fifo.peek().is_none());

        let entry = FotCountEntry { bd_id: 7, write_count: 42, last_in_task: true };
        fifo.push(entry);
        let peeked = fifo.peek().unwrap();
        assert_eq!(peeked.bd_id, 7);
        assert_eq!(peeked.write_count, 42);
        assert!(peeked.last_in_task);

        // Peek doesn't remove
        assert_eq!(fifo.len(), 1);
    }

    #[test]
    fn fot_fifo_clear() {
        let mut fifo = FotCountFifo::new();
        for i in 0..5 {
            fifo.push(FotCountEntry { bd_id: i, write_count: 0, last_in_task: false });
        }
        assert_eq!(fifo.len(), 5);

        fifo.clear();
        assert!(fifo.is_empty());
        assert_eq!(fifo.len(), 0);
        assert_eq!(fifo.pop_register(), 0);
    }

    #[test]
    fn fot_fifo_capacity() {
        let fifo = FotCountFifo::with_capacity(8);
        assert_eq!(fifo.capacity(), 8);
    }

    #[test]
    fn fot_fifo_default() {
        let fifo = FotCountFifo::default();
        assert!(fifo.is_empty());
        assert_eq!(fifo.capacity(), FOT_COUNT_FIFO_DEFAULT_CAPACITY);
    }

    #[test]
    fn fot_pop_after_empty_still_returns_zero() {
        let mut fifo = FotCountFifo::new();
        fifo.push(FotCountEntry { bd_id: 0, write_count: 1, last_in_task: true });
        fifo.pop_register(); // consume the one entry
        assert_eq!(fifo.pop_register(), 0); // now empty
        assert_eq!(fifo.pop_register(), 0); // still empty
    }
}
