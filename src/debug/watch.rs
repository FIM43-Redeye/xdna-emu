//! Memory watch mechanism driven by the `XDNA_EMU_WATCH` environment variable.
//!
//! When `XDNA_EMU_WATCH` is set, every memory access (core loads/stores, DMA
//! reads/writes) that touches a watched address range is logged at INFO level.
//! When the variable is not set there is zero runtime overhead: `is_watched`
//! returns immediately because the `OnceLock` is empty.
//!
//! # Configuration
//!
//! ```text
//! XDNA_EMU_WATCH=0xC000:40,0x400:4
//! ```
//!
//! Each entry is `address:bytes` where `address` is a hex value (0x prefix
//! optional) and `bytes` is the number of bytes to watch (defaults to 4 if
//! omitted).  Multiple ranges are comma-separated.
//!
//! # Log format
//!
//! ```text
//! [WATCH] cycle=283 CORE-LD  pc=0x1A0 addr=0x0C000 value=0x00000001 -> ScalarReg(24)
//! [WATCH] cycle=283 CORE-ST  pc=0x1B4 addr=0x00400 value=0x00000005
//! [WATCH] cycle=285 DMA-WR   tile=(0,2) addr=0x0C000 value=0x00000004 ch=S2MM0
//! [WATCH] cycle=285 DMA-RD   tile=(0,2) addr=0x00400 value=0x00000005 ch=MM2S2
//! ```

use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// WatchRange
// ---------------------------------------------------------------------------

/// A contiguous address range to watch for memory accesses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WatchRange {
    /// First byte address in the range (inclusive).
    pub start: u64,
    /// Number of bytes in the range.
    pub len: usize,
}

impl WatchRange {
    /// Returns `true` if any byte in `[addr, addr+size)` falls within this range.
    #[inline]
    pub fn overlaps(&self, addr: u64, size: usize) -> bool {
        // Access interval: [addr, addr+size)
        // Range interval:  [self.start, self.start+self.len)
        // Overlap when: addr < range_end && access_end > range_start
        let range_end = self.start.saturating_add(self.len as u64);
        let access_end = addr.saturating_add(size as u64);
        addr < range_end && access_end > self.start
    }
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Parses a comma-separated list of `address:bytes` watch specifications.
///
/// Hex addresses may include the `0x`/`0X` prefix or be bare hex.
/// If `:bytes` is omitted the default width is 4.
/// Invalid entries are skipped with a `warn!` log message.
pub fn parse_watch_env(value: &str) -> Vec<WatchRange> {
    let mut ranges = Vec::new();

    for entry in value.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }

        let (addr_str, len) = if let Some((a, b)) = entry.split_once(':') {
            let b_str = b.trim();
            match b_str.parse::<usize>() {
                Ok(n) => (a.trim(), n),
                Err(_) => {
                    log::warn!("[WATCH] invalid byte count in watch spec {:?}, skipping", entry);
                    continue;
                }
            }
        } else {
            (entry, 4)
        };

        // Strip 0x/0X prefix for u64::from_str_radix
        let hex = addr_str.trim_start_matches("0x").trim_start_matches("0X");
        match u64::from_str_radix(hex, 16) {
            Ok(start) => ranges.push(WatchRange { start, len }),
            Err(_) => {
                log::warn!("[WATCH] invalid address in watch spec {:?}, skipping", entry);
            }
        }
    }

    ranges
}

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

/// Global watch list, populated once by `init()`.
///
/// `None` value inside OnceLock means no watches were configured, which lets
/// `is_watched` short-circuit without any allocation.
static WATCHES: OnceLock<Vec<WatchRange>> = OnceLock::new();

/// Initialise the watch list from the `XDNA_EMU_WATCH` environment variable.
///
/// Safe to call multiple times -- the `OnceLock` ensures only the first call
/// has effect.  Should be called early in emulator startup so that all
/// subsequent memory accesses are instrumented from the beginning.
pub fn init() {
    WATCHES.get_or_init(|| {
        let value = match std::env::var("XDNA_EMU_WATCH") {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };

        let ranges = parse_watch_env(&value);
        if !ranges.is_empty() {
            for r in &ranges {
                log::info!(
                    "[WATCH] watching {:?} bytes starting at 0x{:X}",
                    r.len,
                    r.start
                );
            }
        }
        ranges
    });
}

/// Returns `true` if any configured watch range overlaps `[addr, addr+len)`.
///
/// Returns `false` immediately when no watches have been configured (zero
/// overhead path).
#[inline]
pub fn is_watched(addr: u64, len: usize) -> bool {
    WATCHES
        .get()
        .map(|watches| watches.iter().any(|w| w.overlaps(addr, len)))
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Log helpers
// ---------------------------------------------------------------------------

/// Log a core load that hit a watch range.
///
/// ```text
/// [WATCH] cycle=283 CORE-LD  pc=0x1A0 addr=0x0C000 value=0x00000001 -> ScalarReg(24)
/// ```
pub fn log_core_load(cycle: u64, pc: u32, addr: u64, value: u32, dest: &str) {
    log::info!(
        "[WATCH] cycle={} CORE-LD  pc=0x{:X} addr=0x{:05X} value=0x{:08X} -> {}",
        cycle,
        pc,
        addr,
        value,
        dest
    );
}

/// Log a core store that hit a watch range.
///
/// ```text
/// [WATCH] cycle=283 CORE-ST  pc=0x1B4 addr=0x00400 value=0x00000005
/// ```
pub fn log_core_store(cycle: u64, pc: u32, addr: u64, value: u32) {
    log::info!(
        "[WATCH] cycle={} CORE-ST  pc=0x{:X} addr=0x{:05X} value=0x{:08X}",
        cycle,
        pc,
        addr,
        value
    );
}

/// Log a DMA write that hit a watch range.
///
/// ```text
/// [WATCH] cycle=285 DMA-WR   tile=(0,2) addr=0x0C000 value=0x00000004 ch=S2MM0
/// ```
pub fn log_dma_write(cycle: u64, col: u8, row: u8, addr: u64, value: u32, channel: &str) {
    log::info!(
        "[WATCH] cycle={} DMA-WR   tile=({},{}) addr=0x{:05X} value=0x{:08X} ch={}",
        cycle,
        col,
        row,
        addr,
        value,
        channel
    );
}

/// Log a DMA read that hit a watch range.
///
/// ```text
/// [WATCH] cycle=285 DMA-RD   tile=(0,2) addr=0x00400 value=0x00000005 ch=MM2S2
/// ```
pub fn log_dma_read(cycle: u64, col: u8, row: u8, addr: u64, value: u32, channel: &str) {
    log::info!(
        "[WATCH] cycle={} DMA-RD   tile=({},{}) addr=0x{:05X} value=0x{:08X} ch={}",
        cycle,
        col,
        row,
        addr,
        value,
        channel
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create a WatchRange from literal values.
    fn range(start: u64, len: usize) -> WatchRange {
        WatchRange { start, len }
    }

    #[test]
    fn test_overlap_exact() {
        // Range covers [0xC000, 0xC028) (40 bytes).
        let r = range(0xC000, 40);

        // Exact start address.
        assert!(r.overlaps(0xC000, 4), "start of range should overlap");
        // Last word (bytes 36-39 inclusive).
        assert!(r.overlaps(0xC024, 4), "last word in range should overlap");
        // One word just past the end -- should NOT overlap.
        assert!(!r.overlaps(0xC028, 4), "first word past end should not overlap");
        // One word just before the start -- should NOT overlap.
        assert!(!r.overlaps(0xBFFC, 4), "word before start should not overlap");
    }

    #[test]
    fn test_overlap_partial() {
        // Range [0xC000, 0xC028).
        let r = range(0xC000, 40);

        // Access straddles start: [0xBFFE, 0xC002) -- 4 bytes across boundary.
        assert!(r.overlaps(0xBFFE, 4), "access straddling range start should overlap");
        // Access straddles end: [0xC026, 0xC02A) -- 4 bytes across boundary.
        assert!(r.overlaps(0xC026, 4), "access straddling range end should overlap");
    }

    #[test]
    fn test_parse_basic() {
        let ranges = parse_watch_env("0xC000:40");
        assert_eq!(ranges, vec![range(0xC000, 40)]);
    }

    #[test]
    fn test_parse_no_prefix() {
        let ranges = parse_watch_env("C000:40");
        assert_eq!(ranges, vec![range(0xC000, 40)]);
    }

    #[test]
    fn test_parse_default_len() {
        let ranges = parse_watch_env("0xC000");
        assert_eq!(ranges, vec![range(0xC000, 4)]);
    }

    #[test]
    fn test_parse_multiple() {
        let ranges = parse_watch_env("0xC000:40,0x400:4,0x1000:16");
        assert_eq!(
            ranges,
            vec![range(0xC000, 40), range(0x400, 4), range(0x1000, 16)]
        );
    }

    #[test]
    fn test_parse_empty_and_whitespace() {
        // Leading/trailing whitespace per entry and empty entries should be tolerated.
        let ranges = parse_watch_env("  0xC000:40 , , 0x400:4  ");
        assert_eq!(ranges, vec![range(0xC000, 40), range(0x400, 4)]);
    }

    #[test]
    fn test_parse_invalid_skipped() {
        // "ZZZZ" is not valid hex -- that entry should be skipped.
        let ranges = parse_watch_env("0xC000:40,ZZZZ:4,0x400:4");
        assert_eq!(ranges, vec![range(0xC000, 40), range(0x400, 4)]);
    }
}
